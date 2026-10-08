"""Planar rigid-body skid-steer surrogate of the Gazebo plant.

Pure numpy. Same reset/step/close contract as KinematicBridge, but with
inertia and lateral skid; parameters start at the SDF values and are fitted to
Gazebo by tools/fit_surrogate.py. State is (x, y, th, vx, vy, r) plus four
wheel speeds following the command with lag tau_w. Each wheel's contact force
is linear in slip with compliance c up to mu N_i, clipped per axis (or on an
ellipse with friction="ellipse"); N_i = m g / 4. Integrated with semi-implicit
Euler at h <= h_max because the force law is stiff. surface_fn(x, y) -> mu
caps each wheel's mu1/mu2. Readings mimic GazeboBridge: true pose, actual
wheel speeds, IMU (r, ax, ay) and (v, omega) from wheel odometry.
"""

import json
from dataclasses import asdict, dataclass, fields
from typing import Callable, Optional

import numpy as np

from .bridge import StateReading
from .slip_model import WHEEL_OFFSETS

G = 9.81


@dataclass
class DynamicParams:
    """Plant parameters. Defaults are the SIMULATED robot's, from the files below.

    MASS AND YAW INERTIA (provenance, 2026-10-08). Computed, not estimated, by
    ``tools/urdf_planar_inertia.py`` from the URDF that Gazebo spawns
    (``xacro src/scout_ros2/scout_description/urdf/scout_mini.xacro sim:=true
    sim_sensors:=false``). It composes the joint origins, rotates each link's
    inertia into the base frame and adds the parallel-axis term about the
    vertical axis through the CoM:

    ===================  =======  ==============================================
    link                 mass kg  Izz contribution (own + m d^2) [kg m^2]
    ===================  =======  ==============================================
    base_link            10       0.2667  (scout_mini.xacro)
    box_link              5       0.0542  (scout_mini.xacro, on the axis)
    d435_camera_link      0.072   0.0069  (realsense2_description _d435 macro)
    monitor_link          0.1     0.0130  (scout_mini.xacro)
    4 x *_wheel_link      3 each  4 x ~0.298 = 1.192  (wheel.xacro; the
                                  wheels at (+-0.232, +-0.208) m are 78 %)
    total                27.172   1.5332
    ===================  =======  ==============================================

    CoM is 0.2 mm from base_link horizontally (3.4 cm below it), so the wheel
    offsets about the base origin are offsets about the CoM. These are the
    URDF's numbers -- the SIM plant -- not the real Scout Mini, which has never
    been weighed here. ``izz`` is in FIT_KEYS: a fitted file overrides it, and
    the URDF value is the prior to compare the fit against. ``mass`` is NOT
    fitted (with g it sets the normal loads; only force/mass ratios are
    identifiable from pose data, and the friction terms already carry them).
    """

    # Starting values = scout_mini.xacro / wheel.xacro / rl_corrector.world.
    mass: float = 27.172         # URDF total, see the class docstring
    izz: float = 1.5332          # URDF, about the vertical axis through the CoM
    wheel_radius: float = 0.08
    mu1: float = 1.0             # min(wheel 200, ground 1.0)
    mu2: float = 0.45            # min(wheel 0.45, ground 1.0)
    c1: float = 0.01             # slip1 [m/s per N]
    c2: float = 0.01             # slip2
    tau_w: float = 0.01          # wheel speed lag [s]
    friction: str = "pyramid"    # or "ellipse"
    h_max: float = 0.002         # [s] integration substep ceiling (stiff force law)

    FIT_KEYS = ("izz", "mu1", "mu2", "c1", "c2", "tau_w")

    @classmethod
    def load(cls, path: str) -> "DynamicParams":
        with open(path) as fh:
            d = json.load(fh)
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def save(self, path: str, **extra) -> None:
        with open(path, "w") as fh:
            json.dump({**asdict(self), **extra}, fh, indent=2)


_A = np.array([o[0] for o in WHEEL_OFFSETS])   # [fl, rl, fr, rr] forward offsets
_B = np.array([o[1] for o in WHEEL_OFFSETS])   # lateral offsets (+ = left)


def simulate(p: DynamicParams, state0: np.ndarray, om0: np.ndarray, cmds: np.ndarray,
             dt: float, mu_fn: Optional[Callable] = None, record_every: int = 1):
    """Batched rollout.

    state0 (B, 6) = (x, y, th, vx, vy, r); om0 (B, 4) wheel speeds;
    cmds (B, T, 4) wheel commands held for `dt` each. Returns
    ``states (B, T+1, 6)``, ``oms (B, T+1, 4)``, ``acc (B, T+1, 2)`` (body
    ax, ay at the end of each step). `mu_fn(x, y) -> (B, 4)` ground mu under
    each wheel, or None for uniform ground (mu 1).
    """
    s = np.array(state0, float, copy=True)
    om = np.array(om0, float, copy=True)
    B, T, _ = cmds.shape
    n = max(1, int(np.ceil(dt / p.h_max - 1e-9)))
    h = dt / n
    Nz = p.mass * G / 4.0
    states = np.empty((B, T + 1, 6)); states[:, 0] = s
    oms = np.empty((B, T + 1, 4)); oms[:, 0] = om
    acc = np.zeros((B, T + 1, 2))
    alpha = 1.0 if p.tau_w <= 0 else 1.0 - np.exp(-h / p.tau_w)
    for k in range(T):
        cmd = cmds[:, k]
        for _ in range(n):
            om += alpha * (cmd - om)
            vx, vy, r = s[:, 3:4], s[:, 4:5], s[:, 5:6]
            sx = (vx - r * _B) - p.wheel_radius * om          # (B, 4)
            sy = vy + r * _A
            if mu_fn is None:
                lim1, lim2 = p.mu1 * Nz, p.mu2 * Nz
            else:
                mu = mu_fn(s[:, 0], s[:, 1], s[:, 2])         # (B, 4)
                lim1 = np.minimum(p.mu1, mu) * Nz
                lim2 = np.minimum(p.mu2, mu) * Nz
            fx = -sx / p.c1
            fy = -sy / p.c2
            if p.friction == "ellipse":
                q = np.sqrt((fx / lim1) ** 2 + (fy / lim2) ** 2)
                sc = np.where(q > 1.0, 1.0 / np.maximum(q, 1e-12), 1.0)
                fx, fy = fx * sc, fy * sc
            else:
                fx = np.clip(fx, -lim1, lim1)
                fy = np.clip(fy, -lim2, lim2)
            ax = fx.sum(1) / p.mass
            ay = fy.sum(1) / p.mass
            dr = (_A * fy - _B * fx).sum(1) / p.izz
            # semi-implicit: velocities first, then pose with the new ones
            vxn = s[:, 3] + h * (ax + s[:, 5] * s[:, 4])
            vyn = s[:, 4] + h * (ay - s[:, 5] * s[:, 3])
            rn = s[:, 5] + h * dr
            s[:, 3], s[:, 4], s[:, 5] = vxn, vyn, rn
            c, si = np.cos(s[:, 2]), np.sin(s[:, 2])
            s[:, 0] += h * (c * vxn - si * vyn)
            s[:, 1] += h * (si * vxn + c * vyn)
            s[:, 2] += h * rn
        states[:, k + 1] = s
        oms[:, k + 1] = om
        acc[:, k + 1, 0], acc[:, k + 1, 1] = ax, ay
    return states, oms, acc


class DynamicBridge:
    """Bridge-contract wrapper around `simulate` for one robot."""

    def __init__(self, cfg, params: Optional[DynamicParams] = None,
                 surface_fn: Optional[Callable[[float, float], float]] = None,
                 surface_from_terrain: bool = False) -> None:
        self.cfg = cfg
        self.p = params or DynamicParams()
        self.surface_fn = surface_fn
        self.surface_from_terrain = surface_from_terrain
        self._s = np.zeros((1, 6))
        self._om = np.zeros((1, 4))

    def _mu_fn(self):
        if self.surface_fn is None:
            return None
        f = self.surface_fn

        def mu_fn(x, y, th):
            out = np.empty((x.shape[0], 4))
            for j in range(x.shape[0]):
                c, s = np.cos(th[j]), np.sin(th[j])
                out[j] = [f(x[j] + c * a - s * b, y[j] + s * a + c * b) for a, b in WHEEL_OFFSETS]
            return out
        return mu_fn

    def _reading(self, acc=(0.0, 0.0)) -> StateReading:
        x, y, th, _, _, r = self._s[0]
        om = self._om[0]
        v, omega = self.cfg.wheels_to_body(0.5 * (om[0] + om[1]), 0.5 * (om[2] + om[3]))
        return StateReading((float(x), float(y), float(th)), float(v), float(omega),
                            [float(w) for w in om], False,
                            imu=(float(r), float(acc[0]), float(acc[1])))

    def reset(self, start_pose, terrain=None) -> StateReading:
        self._s = np.zeros((1, 6)); self._s[0, :3] = [float(v) for v in start_pose]
        self._om = np.zeros((1, 4))
        if self.surface_from_terrain and isinstance(terrain, (list, tuple)):
            from . import slip_model
            self.surface_fn = slip_model.patches_to_surface_fn(terrain)
        return self._reading()

    def step(self, wheels, dt: float) -> StateReading:
        cmds = np.asarray(wheels, float).reshape(1, 1, 4)
        states, oms, acc = simulate(self.p, self._s, self._om, cmds, dt, self._mu_fn())
        self._s, self._om = states[:, -1].copy(), oms[:, -1].copy()
        return self._reading(acc[0, -1])

    def close(self) -> None:
        pass
