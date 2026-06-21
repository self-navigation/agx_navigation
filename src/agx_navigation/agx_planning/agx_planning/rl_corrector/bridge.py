"""Simulation-bridge interface used by the Gym env.

The env talks only to a Bridge, so its logic can be tested on the pure
KinematicBridge and run unchanged on GazeboBridge.

A Bridge must implement:
    reset(start_pose, terrain=None) -> StateReading
    step(wheels, dt)                -> StateReading
    close()                         -> None
where `wheels` is the 4-element [fl, rl, fr, rr] command.
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
