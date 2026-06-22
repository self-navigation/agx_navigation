"""Simulation-bridge interface shared by the Gym env.

The env never talks to Gazebo (or anything) directly; it talks to a Bridge.
This keeps the env's MDP logic testable with a pure KinematicBridge and lets the
real GazeboBridge be swapped in unchanged.

A Bridge must implement:
    reset(start_pose, terrain=None) -> StateReading
    step(wheels, dt)                -> StateReading
    close()                         -> None
where `wheels` is the 4-element [fl, rl, fr, rr] command and `start_pose`/`terrain`
configure one episode (terrain is None until Phase 3).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class StateReading:
    """One post-step snapshot of the robot state the env needs."""

    pose: Tuple[float, float, float]      # (x, y, theta) in the planning frame
    v: float                              # measured body linear velocity [m/s]
    omega: float                          # measured body yaw rate [rad/s]
    wheel_speeds: Optional[List[float]]   # measured [fl, rl, fr, rr] or None
    contact: bool                         # physical contact (collision) flag
    # IMU reading (gyro_z, accel_x, accel_y) in the body frame, or None if unused.
    # Defaulted so bridges that don't surface it (KinematicBridge) stay unchanged.
    imu: Optional[Tuple[float, float, float]] = None
