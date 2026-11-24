"""Declared ROS-free execution models and conservative motion bounds.

Command replay uses ideal, piecewise constant wheel setpoints. It does not
model an actuator ramp. Nominal replay follows the saved BVP polynomials.
Neither model is a measured physical execution result.
"""

from dataclasses import dataclass
import math

import numpy as np
from numpy.polynomial import Polynomial

from .validity import Footprint


@dataclass(frozen=True)
class ReplayTrajectory:
    times: np.ndarray
    states: np.ndarray
    motion_bounds: np.ndarray

    def __post_init__(self):
        for name in ("times", "states", "motion_bounds"):
            array = np.asarray(getattr(self, name), dtype=float)
            if not np.isfinite(array).all():
                raise ValueError("replay arrays must be finite")
            array = np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape)
            object.__setattr__(self, name, array)
        n = len(self.times)
        if (n < 1 or self.times.shape != (n,) or self.states.shape != (n, 5)
                or self.motion_bounds.shape != (n - 1,)
                or np.any(np.diff(self.times) <= 0) or np.any(self.motion_bounds < 0)):
            raise ValueError("replay requires increasing times and N-1 motion bounds")


def command_step(state, wheels, dt, c_v, c_w):
    """Exact constant-wheel skid-steer arc with unwrapped heading."""
    state, wheels = np.asarray(state, float), np.asarray(wheels, float)
    if (state.shape != (5,) or wheels.shape != (2,)
            or not np.isfinite(state).all() or not np.isfinite(wheels).all()
            or not all(math.isfinite(x) for x in (dt, c_v, c_w))
            or dt < 0 or min(c_v, c_w) <= 0):
        raise ValueError("invalid command replay input")
    v = c_v * wheels.sum()
    omega = c_w * (wheels[1] - wheels[0])
    angle = omega * dt
    distance = v * dt * np.sinc(angle / (2 * math.pi))
    midpoint = state[2] + angle / 2
    return np.array([state[0] + distance * math.cos(midpoint),
                     state[1] + distance * math.sin(midpoint),
                     state[2] + angle, *wheels])


def _polynomial_motion(segment, left, right, radius):
    """Integrate a coefficient bound on |x'|+|y'|+radius*|theta'|.

    Split at every spline knot. Translating each derivative to the interval
    start avoids bounding a short interval using the whole spline horizon.
    This bounds the stored polynomial, including its interpolation residual.
    """
    total = 0.0
    cuts = np.r_[left, segment.knots[(segment.knots > left) & (segment.knots < right)], right]
    for a, b in zip(cuts[:-1], cuts[1:]):
        index = min(np.searchsorted(segment.knots, a, side="right") - 1,
                    len(segment.knots) - 2)
        offset, width = a - segment.knots[index], b - a
        for channel, weight in ((0, 1.0), (1, 1.0), (2, radius)):
            derivative = Polynomial(segment.coefficients[::-1, index, channel]).deriv()
            shifted = derivative(Polynomial([offset, 1.0])).coef
            powers = np.arange(1, len(shifted) + 1)
            total += weight * np.sum(np.abs(shifted) * width ** powers / powers)
    return float(total + 128 * np.finfo(float).eps * (1 + total))


def replay_plan(plan, c_v, c_w, *, mode="command", max_step=0.02,
                initial_state=None, footprint=Footprint(), sample_times=()):
    """Replay all N stored command intervals; never infer missing raw states."""
    if mode not in ("command", "nominal"):
        raise ValueError("execution mode must be command or nominal")
    if not math.isfinite(max_step) or max_step <= 0:
        raise ValueError("max_step must be positive")
    extra = np.asarray(sample_times, float)
    if (extra.ndim != 1 or not np.isfinite(extra).all()
            or np.any(extra < 0) or np.any(extra > plan.times[-1])):
        raise ValueError("extra sample times must be within the committed plan")
    state = np.array(plan.raw_states[0] if initial_state is None else initial_state, float)
    if state.shape != (5,) or not np.isfinite(state).all():
        raise ValueError("initial state must contain five finite values")
    if mode == "nominal" and (not np.allclose(state, plan.raw_states[0], rtol=0, atol=1e-6)
                               or (len(plan.commands) and not plan.segments)):
        raise ValueError("nominal replay requires matching raw states and stored splines")
    times, states, motion = [0.0], [state], []
    starts = np.array([segment.start_time for segment in plan.segments])
    for index, (left, right) in enumerate(zip(plan.times[:-1], plan.times[1:])):
        cuts = np.unique(np.r_[left, extra[(extra > left) & (extra < right)], right])
        steps = [left]
        for a, b in zip(cuts[:-1], cuts[1:]):
            count = max(1, math.ceil((b - a) / max_step))
            steps.extend(np.linspace(a, b, count + 1)[1:])
        for a, b in zip(steps[:-1], steps[1:]):
            if mode == "command":
                wheels = plan.commands[index]
                state = command_step(state, wheels, b - a, c_v, c_w)
                bound = (b - a) * (abs(c_v * wheels.sum())
                         + footprint.radius * abs(c_w * (wheels[1] - wheels[0])))
            else:
                segment = plan.segments[np.searchsorted(starts, a + 1e-10, side="right") - 1]
                local_a = max(0.0, a - segment.start_time)
                local_b = min(segment.duration, b - segment.start_time)
                state = segment.evaluate(local_b)[:5]
                bound = _polynomial_motion(segment, local_a, local_b, footprint.radius)
            times.append(float(b))
            states.append(state)
            motion.append(bound)
    return ReplayTrajectory(times, states, motion)
