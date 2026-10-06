"""Back-off-and-face recovery for a robot held at a wall. ROS-free.

The re-join solve ignores walls, so a robot stopped facing a wall keeps
getting re-joins that the wall gate rejects. backoff_and_face builds a short
reverse followed by a turn in place toward the re-join target, as
PlaybackSamples predicted with ideal kinematics, which the corrector tracks
like any overlay. The next re-join starts from the measured state, so an
under-rotated turn is harmless.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..runtime_corrector.trajectory_buffer import PlaybackSample


def backoff_and_face(
    pose: Sequence[float],
    target_xy: Sequence[float],
    clearance: Callable[[float, float, float], float],
    to_wheels: Callable[[float, float], Tuple[float, float]],
    dt: float,
    v_back: float = 0.2,
    back_max: float = 0.4,
    back_goal: float = 0.35,
    omega: float = 0.6,
    contact: float = 0.03,
) -> Optional[List[PlaybackSample]]:
    """Reverse straight, then turn in place toward ``target_xy``.

    The reverse stops at ``back_max`` metres, once the hull clearance reaches
    ``back_goal``, or as soon as reversing would bring the hull closer to a
    wall (a wall behind). The turn takes the short way. Returns None when
    neither part moves the robot, or when any sample would come within
    ``contact`` of a wall while being closer than at the start.
    ``clearance(x, y, yaw)`` is the signed hull clearance and
    ``to_wheels(v, omega)`` the chassis inverse kinematics. The last sample
    commands zero wheels.
    """
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    c0 = clearance(x, y, th)
    out: List[PlaybackSample] = []
    wl, wr = to_wheels(-v_back, 0.0)
    step = v_back * dt
    c = c0
    moved = 0.0
    while moved + step <= back_max + 1e-9 and c < back_goal:
        nx, ny = x - step * math.cos(th), y - step * math.sin(th)
        nc = clearance(nx, ny, th)
        if nc < c - 1e-6:
            break
        out.append(PlaybackSample(left=wl, right=wr, pose=(x, y, th)))
        x, y, c = nx, ny, nc
        moved += step

    h = math.atan2(float(target_xy[1]) - y, float(target_xy[0]) - x)
    dth = (h - th + math.pi) % (2.0 * math.pi) - math.pi
    n_turn = int(math.ceil(abs(dth) / (omega * dt) - 1e-9))
    if n_turn:
        w = math.copysign(omega, dth)
        tl, tr = to_wheels(0.0, w)
        for i in range(n_turn):
            a = th + dth * i / n_turn
            out.append(PlaybackSample(left=tl, right=tr, pose=(x, y, a)))
        th = th + dth
    if not out:
        return None
    out.append(PlaybackSample(left=0.0, right=0.0, pose=(x, y, th)))
    cs = np.array([clearance(*s.pose) for s in out])
    if np.any(cs < np.minimum(contact, c0) - 0.01):
        return None
    return out
