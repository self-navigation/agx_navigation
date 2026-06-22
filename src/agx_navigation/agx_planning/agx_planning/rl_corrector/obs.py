"""Observation construction. Pure (numpy only).

The tracking error is expressed in the PLANNED-heading frame (along/cross-track),
not absolute map coordinates, so the policy is position-invariant and generalizes
anywhere on the map. This exact builder is used both by the training env and by
the deployed _correct(), so the observation a policy sees at inference matches
what it was trained on.

Observation layout (fixed order):
  [ e_along, e_cross, e_heading,                 # tracking error (path frame)
    e_along_dot, e_cross_dot, e_heading_dot,     # error rates
    cmd_left, cmd_right,                          # nominal commands (normalized)
    v_meas, omega_meas,                           # measured body twist (normalized)
   (prev_coeff - 1) * action_dim   if use_prev_coeff,
    imu_gyro_z, imu_ax, imu_ay     if use_imu,
    wheel_speeds * 4               if use_wheel_speeds,
    costates * 5                   if use_costates ]

The IMU block (true yaw rate + body-frame longitudinal/lateral acceleration) is a
SLIP-OBSERVING signal available on the real robot: unlike v_meas/omega_meas (which
are wheel-derived and blind to slip), the gyro/accelerometer measure the body's
actual motion, so the policy can see the gap between commanded/kinematic motion and
what really happened -- the very thing it corrects.
"""

from typing import Optional, Tuple

import numpy as np


def wrap_to_pi(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def tracking_error(planned_pose, actual_pose) -> np.ndarray:
    """(along, cross, heading) error of actual vs planned, rotated into the
    planned-heading frame. along = ahead(+)/behind(-); cross = left(+)/right(-)."""
    px, py, pth = planned_pose
    ax, ay, ath = actual_pose
    dx = ax - px
    dy = ay - py
    c = np.cos(pth)
    s = np.sin(pth)
    e_along = c * dx + s * dy
    e_cross = -s * dx + c * dy
    e_heading = wrap_to_pi(ath - pth)
    return np.array([e_along, e_cross, e_heading], dtype=float)


def observation_dim(cfg) -> int:
    n = 3 + 3 + 2 + 2
    if cfg.use_prev_coeff:
        n += cfg.action_dim
    if cfg.use_imu:
        n += 3
    if cfg.use_wheel_speeds:
        n += 4
    if cfg.use_costates:
        n += 5
    return n


def build_observation(
    cfg,
    planned_pose,
    actual_pose,
    prev_err,
    cmd_left: float,
    cmd_right: float,
    v_meas: float,
    omega_meas: float,
    prev_coeff=None,
    imu=None,
    wheel_speeds=None,
    costates=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the (normalized, float32) observation vector.

    `prev_err` is the previous step's tracking error (None on the first step ->
    zero rates). Returns (observation, current_error); the caller stores the
    returned error as next step's prev_err.
    """
    err = tracking_error(planned_pose, actual_pose)
    if prev_err is None:
        rate = np.zeros(3)
    else:
        rate = (err - np.asarray(prev_err, dtype=float)) / cfg.control_dt

    parts = [
        err[0] / cfg.pos_err_norm,
        err[1] / cfg.pos_err_norm,
        err[2] / np.pi,
        rate[0] / cfg.rate_norm,
        rate[1] / cfg.rate_norm,
        rate[2] / cfg.rate_norm,
        cmd_left / cfg.wheel_cmd_max,
        cmd_right / cfg.wheel_cmd_max,
        v_meas / cfg.twist_v_norm,
        omega_meas / cfg.twist_w_norm,
    ]

    if cfg.use_prev_coeff:
        pc = (
            np.ones(cfg.action_dim)
            if prev_coeff is None
            else np.asarray(prev_coeff, dtype=float).ravel()
        )
        parts.extend((pc - 1.0).tolist())  # centered at identity

    if cfg.use_imu:
        # (gyro_z, ax, ay): yaw rate normalized like omega, body accel by its own
        # scale. Missing reading -> zeros (same fail-safe convention as the rest).
        im = np.zeros(3) if imu is None else np.asarray(imu, dtype=float).ravel()
        parts.extend([
            im[0] / cfg.imu_gyro_norm,
            im[1] / cfg.imu_accel_norm,
            im[2] / cfg.imu_accel_norm,
        ])

    if cfg.use_wheel_speeds:
        ws = (
            np.zeros(4)
            if wheel_speeds is None
            else np.asarray(wheel_speeds, dtype=float).ravel()
        )
        parts.extend((ws / cfg.wheel_cmd_max).tolist())

    if cfg.use_costates:
        cs = (
            np.zeros(5)
            if costates is None
            else np.asarray(costates, dtype=float).ravel()
        )
        parts.extend((cs / cfg.costate_norm).tolist())

    return np.asarray(parts, dtype=np.float32), err
