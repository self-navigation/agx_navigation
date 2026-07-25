"""Playback cursor tests for TrajectoryBuffer.

The distinction under test is the one that decides whether the corrector reaches
its goal: a TIME-indexed cursor drains one sample per tick no matter where the
robot is, so any speed spent on corrections is lost distance; a PROGRESS-indexed
cursor still steps per tick but refuses to lead the robot by more than
`max_lead` samples. See the module docstring in trajectory_buffer.py for the
measurements behind that, and for why slaving the cursor to progress outright
deadlocks a plan that starts from rest.
"""

import math

from agx_planning.runtime_corrector.trajectory_buffer import TrajectoryBuffer


class Chunk:
    """Minimal stand-in for one PlanToGoal feedback chunk."""

    def __init__(self, poses, left=None, right=None):
        self.pose_x = [p[0] for p in poses]
        self.pose_y = [p[1] for p in poses]
        self.pose_theta = [p[2] if len(p) > 2 else 0.0 for p in poses]
        n = len(poses)
        self.wheel_left = left if left is not None else [1.0] * n
        self.wheel_right = right if right is not None else [1.0] * n


def straight_line(n, dx=0.1):
    """A plan running along +x, one sample every `dx` metres."""
    return Chunk([(i * dx, 0.0, 0.0) for i in range(n)])


def make(chunks, dt=0.1, max_skip=20, max_lead=10, result=True):
    buf = TrajectoryBuffer(dt, max_skip=max_skip, max_lead=max_lead)
    buf.reset(traj_id=1, dt=dt)
    for i, c in enumerate(chunks):
        buf.add_chunk(i, c)
    if result:
        buf.mark_result(True)
    return buf


# ---------------------------------------------------------------- time mode


def test_time_mode_serves_one_sample_per_call():
    buf = make([straight_line(5)])
    xs = []
    while True:
        s = buf.advance()
        if s is None:
            break
        xs.append(s.pose[0])
    assert xs == [0.0, 0.1, 0.2, 0.30000000000000004, 0.4]
    assert buf.is_done()


def test_time_mode_drains_regardless_of_where_the_robot_is():
    """The defect this whole change exists for: the cursor never consults the
    robot, so a robot that has not moved still runs out of trajectory."""
    buf = make([straight_line(5)])
    for _ in range(5):
        assert buf.advance() is not None
    assert buf.is_done()


# ------------------------------------------------------------ progress mode


def test_progress_mode_leads_a_stationary_robot_off_the_line():
    """The regression that parked the robot on its spawn point: the plan starts
    from rest, so if the cursor is slaved to measured progress, sample 0
    commands zero, the robot never moves, and the cursor never advances. The
    reference MUST be allowed to lead."""
    buf = make([straight_line(20)], max_lead=3)
    seen = [buf.advance(actual_xy=(0.0, 0.0)).pose[0] for _ in range(6)]
    assert seen[0] == 0.0
    assert max(seen) > 0.0, "cursor never left sample 0 -- the fixed point is back"


def test_progress_mode_stops_leading_beyond_max_lead():
    """...but it must not abandon the robot either: with the robot parked, the
    reference runs out to max_lead and then holds."""
    buf = make([straight_line(20)], max_lead=3)
    for _ in range(10):
        s = buf.advance(actual_xy=(0.0, 0.0))
    # The lead is checked BEFORE the step, so the reference settles one sample
    # past max_lead and holds there.
    assert math.isclose(s.pose[0], 0.4, abs_tol=1e-9)
    assert not buf.is_done()


def test_progress_mode_resumes_when_the_robot_catches_up():
    buf = make([straight_line(20)], max_lead=3)
    for _ in range(10):
        buf.advance(actual_xy=(0.0, 0.0))  # pinned at the lead limit
    s = buf.advance(actual_xy=(0.5, 0.0))  # robot arrives
    assert math.isclose(s.pose[0], 0.4, abs_tol=1e-9)
    s = buf.advance(actual_xy=(0.5, 0.0))
    assert math.isclose(s.pose[0], 0.5, abs_tol=1e-9)  # moving again


def test_progress_mode_matches_time_mode_when_the_robot_keeps_up():
    """A robot that tracks the plan should see ordinary time playback."""
    buf = make([straight_line(10)], max_lead=3)
    for i in range(10):
        s = buf.advance(actual_xy=(i * 0.1, 0.0))
        assert math.isclose(s.pose[0], i * 0.1, abs_tol=1e-9)
    assert buf.is_done()


def test_progress_mode_keeps_cross_track_error_visible():
    """The lateral offset must survive projection -- that residual is exactly
    what the corrector is there to remove."""
    buf = make([straight_line(10)], max_lead=3)
    for _ in range(4):
        s = buf.advance(actual_xy=(0.3, 0.25))
    assert math.isclose(s.pose[1], 0.0, abs_tol=1e-9)


def test_progress_projection_never_moves_backwards():
    buf = make([straight_line(20)], max_lead=100)
    for _ in range(8):
        buf.advance(actual_xy=(0.7, 0.0))
    before = buf.advance(actual_xy=(0.0, 0.0))  # robot slid back
    after = buf.advance(actual_xy=(0.0, 0.0))
    assert after.pose[0] >= before.pose[0]


def test_progress_projection_cannot_skip_past_the_window():
    """A plan that loops back near itself must not let the projection teleport."""
    buf = make([straight_line(100)], max_skip=3, max_lead=0)
    for _ in range(3):
        s = buf.advance(actual_xy=(9.9, 0.0))  # far ahead of the projection
    assert s.pose[0] < 1.0  # clamped, not teleported


def test_progress_mode_not_done_before_result_arrives():
    """Reaching the end of what has been RECEIVED is not reaching the end of
    the plan -- without the result those look identical."""
    buf = make([straight_line(5)], result=False, max_lead=3)
    for _ in range(10):
        buf.advance(actual_xy=(0.4, 0.0))
    assert not buf.is_done()


# ------------------------------------------------------------------ chunks


def test_out_of_order_chunks_are_held_until_contiguous():
    buf = TrajectoryBuffer(0.1)
    buf.reset(traj_id=1, dt=0.1)
    buf.add_chunk(1, Chunk([(1.0, 0.0, 0.0)]))  # arrives first
    assert buf.advance() is None  # chunk 0 missing -- hold
    buf.add_chunk(0, Chunk([(0.0, 0.0, 0.0)]))
    assert buf.advance().pose[0] == 0.0
    assert buf.advance().pose[0] == 1.0


def test_hold_is_distinguishable_from_done():
    buf = make([straight_line(2)], result=False)
    buf.advance()
    buf.advance()
    assert buf.advance() is None
    assert not buf.is_done()  # starved, not finished


def test_clear_resets_the_cursor():
    buf = make([straight_line(5)])
    buf.advance()
    buf.clear()
    assert buf.progress == (0, 0)
    assert buf.active_traj_id == -1


def test_polyline_starts_at_the_cursor():
    buf = make([straight_line(5)])
    buf.advance()
    buf.advance()
    poly = buf.build_polyline()
    assert math.isclose(poly[0][0], 0.2, abs_tol=1e-9)
    assert len(poly) == 3


# --------------------------------------------------------- terminal condition


def test_progress_mode_ends_on_the_ROBOT_arriving_not_the_cursor():
    """The gate bounds the reference to max_lead ahead, so ending when the
    CURSOR lands leaves that whole lead unexecuted -- which is most of the
    shortfall the gate exists to remove."""
    buf = make([straight_line(10)], max_lead=3)
    for _ in range(30):
        buf.advance(actual_xy=(0.0, 0.0))  # robot never moves
    assert not buf.is_done(), "finished with the robot still at the start"


def test_progress_mode_holds_the_last_sample_until_the_robot_lands():
    buf = make([straight_line(10)], max_lead=3)
    for _ in range(30):
        s = buf.advance(actual_xy=(0.6, 0.0))
    assert math.isclose(s.pose[0], 0.9, abs_tol=1e-9)  # clamped at the last
    assert not buf.is_done()
    buf.advance(actual_xy=(0.9, 0.0))  # robot arrives
    assert buf.is_done()
