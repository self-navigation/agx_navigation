"""A pure-Python no-slip (optionally slipping) kinematic bridge.

Integrates the same chassis kinematics the nominals are generated from, so an
identity policy on a slip-free bridge tracks the nominal exactly -- the
correctness anchor for the env's MDP wiring and unit tests. An optional per-wheel
slip vector lets it crudely fake terrain without Gazebo (a fast sanity backend),
but it is NOT a physics sim; real training uses GazeboBridge.

SURFACE SLIP (S4, #44). With ``surface_fn`` (``(x, y) -> mu``, e.g. from
``slip_model.patches_to_surface_fn``) the yaw rate follows the measured
chi(mu): ``omega_actual = omega_ideal / chi(mu)``, where ``omega_ideal`` is the
no-slip rate (the cfg's ``wheels_to_body`` already divides by the nominal
``cfg.slip_chi``, so that factor is undone first). At or below the knee
(mu <= 0.45) the yaw gain is ~5 % and forward speed is scaled by an ASSUMED 0.9
(see slip_model). mu is looked up at the robot centre at the start of the step.
The gyro (``imu[0]``) and ``omega`` report the ACTUAL yaw rate, so slip stays
observable; the wheel speeds report what was commanded. With
``surface_from_terrain=True``, a list of patch dicts passed to ``reset`` as
``terrain`` (the GazeboBridge schema) builds ``surface_fn`` for that episode.
With neither, behaviour is byte-identical to the pre-S4 bridge.
"""

from typing import Callable, List, Optional

import numpy as np

from . import slip_model
from .bridge import StateReading


class KinematicBridge:
    def __init__(self, cfg, slip: Optional[List[float]] = None,
                 surface_fn: Optional[Callable[[float, float], float]] = None,
                 surface_from_terrain: bool = False) -> None:
        self.cfg = cfg
        self.surface_fn = surface_fn
        self.surface_from_terrain = surface_from_terrain
        self.last_mu: Optional[float] = None
        # Per-wheel multiplicative slip [fl, rl, fr, rr]; 1.0 == no slip.
        self.slip = None if slip is None else np.asarray(slip, dtype=float)
        self._x = self._y = self._th = 0.0
        self._v_prev = 0.0  # for the synthetic IMU longitudinal accel

    def reset(self, start_pose, terrain=None) -> StateReading:
        self._x, self._y, self._th = (float(v) for v in start_pose)
        self._v_prev = 0.0
        # `terrain` may carry per-wheel slip for a quick fake; ignored otherwise.
        if isinstance(terrain, dict) and "slip" in terrain:
            self.slip = np.asarray(terrain["slip"], dtype=float)
        if self.surface_from_terrain and isinstance(terrain, (list, tuple)):
            self.surface_fn = slip_model.patches_to_surface_fn(terrain)
        # Synthetic IMU at rest (gyro_z, ax, ay) = 0 -- keeps the obs IMU channels
        # consistent with the GazeboBridge so a kinematic-pretrained policy --loads
        # into a Gazebo fine-tune without the IMU block flipping from dead to live.
        return StateReading((self._x, self._y, self._th), 0.0, 0.0, [0.0] * 4, False,
                            imu=(0.0, 0.0, 0.0))

    def step(self, wheels, dt: float) -> StateReading:
        w = np.asarray(wheels, dtype=float)
        if self.slip is not None:
            w = w * self.slip
        # Per-side effective speed = mean of that side's two wheels.
        wl = 0.5 * (w[0] + w[1])
        wr = 0.5 * (w[2] + w[3])
        v, omega = self.cfg.wheels_to_body(wl, wr)
        if self.surface_fn is not None:
            mu = float(self.surface_fn(self._x, self._y))
            self.last_mu = mu
            # cfg's omega already carries the nominal chi; undo it, apply chi(mu).
            omega = omega * self.cfg.slip_chi * slip_model.yaw_gain_of_mu(mu)
            v = v * slip_model.long_factor_of_mu(mu)
        self._x += v * np.cos(self._th) * dt
        self._y += v * np.sin(self._th) * dt
        self._th += omega * dt
        # Synthetic body-frame IMU from the kinematic state: yaw rate = omega,
        # longitudinal accel = dv/dt, lateral accel = centripetal v*omega. Clean
        # (no slip-induced noise), but it gives the policy a non-trivial,
        # correctly-scaled IMU signal during pretraining instead of constant zeros.
        ax = (v - self._v_prev) / dt if dt > 0 else 0.0
        ay = v * omega
        self._v_prev = v
        return StateReading(
            (self._x, self._y, self._th), float(v), float(omega), list(map(float, w)),
            False, imu=(float(omega), float(ax), float(ay)),
        )

    def close(self) -> None:
        pass
