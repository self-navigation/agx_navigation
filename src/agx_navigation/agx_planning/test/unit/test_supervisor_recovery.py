"""Back-off-and-face recovery (supervisor/recovery.py)."""

import math

from agx_planning.supervisor.recovery import backoff_and_face


def wall_ahead(x, y, yaw):
    # A wall at x = 1.0, hull nose 0.3 m ahead of the centre (yaw ignored).
    return 1.0 - x - 0.3


def to_wheels(v, w):
    return (v - 0.2 * w) / 0.08, (v + 0.2 * w) / 0.08


def test_backs_off_then_turns_to_face_the_target():
    s = backoff_and_face((0.65, 0.0, 0.0), (0.0, 5.0), wall_ahead, to_wheels, 0.1)
    assert s is not None
    last = s[-1].pose
    assert wall_ahead(*last) >= 0.35 - 1e-9          # backed off to back_goal
    assert abs(last[0] - 0.35) < 1e-6 and last[1] == 0.0
    assert abs(last[2] - math.atan2(5.0, -0.35)) < 1e-9  # faces the target
    assert s[0].left < 0 and s[0].right < 0           # reverse first
    assert (s[-1].left, s[-1].right) == (0.0, 0.0)


def test_does_not_reverse_into_a_wall_behind():
    def boxed(x, y, yaw):  # walls ahead and behind; reversing gets closer
        return min(1.0 - x - 0.3, x + 1.0 - 0.3) if x < 0.5 else 1.0 - x - 0.3
    s = backoff_and_face((0.0, 0.0, 0.0), (0.0, 5.0), boxed, to_wheels, 0.1)
    assert s is not None and all(p.pose[0] == 0.0 for p in s)  # turn only


def test_nothing_to_do_returns_none():
    s = backoff_and_face((0.0, 0.0, 0.0), (5.0, 0.0), lambda x, y, t: 2.0, to_wheels, 0.1)
    assert s is None
