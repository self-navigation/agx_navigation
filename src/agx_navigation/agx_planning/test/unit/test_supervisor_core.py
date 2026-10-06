"""SupervisorCore: waypoints, per-segment lag, trigger rule, wall scaling."""

import math

import numpy as np
import pytest

from agx_planning.supervisor.core import SupervisorConfig, SupervisorCore
from agx_planning.supervisor.walls import SignedWallDistance


def straight(n=201, length=10.0, dt=0.25):
    x = np.linspace(0, length, n)
    poses = np.c_[x, np.zeros(n), np.zeros(n)]
    return poses, np.arange(n) * dt  # 10 m in 50 s => 0.2 m/s


def test_waypoints_every_5_percent_of_arc_length():
    sup = SupervisorCore()
    poses, t = straight()
    sup.set_plan(poses, t)
    assert sup.n_waypoints == 20
    assert np.allclose(sup.s[sup.wp_index], np.arange(1, 21) * 0.5)


def test_waypoints_ignore_turn_in_place_samples():
    # 100 samples spinning at the origin, then 100 samples driving 10 m.
    spin = np.c_[np.zeros(100), np.zeros(100), np.linspace(0, math.pi, 100)]
    x = np.linspace(0, 10, 101)[1:]
    drive = np.c_[x, np.zeros(100), np.full(100, math.pi)]
    sup = SupervisorCore()
    sup.set_plan(np.vstack([spin, drive]), np.arange(200) * 0.1)
    assert sup.wp_index[0] >= 100  # first waypoint is 0.5 m along, past the spin


def test_on_schedule_robot_never_triggers_and_retimes_at_each_waypoint():
    sup = SupervisorCore()
    poses, t = straight()
    sup.set_plan(poses, t)
    retimes = []
    for k in range(len(poses)):
        d = sup.update(poses[k, 0], 0.0, 0.0, t[k])
        assert not d.trigger, (k, d)
        if d.retime_index is not None:
            retimes.append(d.retime_index)
    assert sup.done
    assert len(retimes) == 20
    assert retimes == sorted(retimes)


def test_lag_trigger_needs_both_doubling_and_distance():
    cfg = SupervisorConfig()
    sup = SupervisorCore(cfg)
    poses, t = straight()
    sup.set_plan(poses, t)
    # Robot moves at half the plan's speed: lag grows within the first segment.
    fired_at = None
    for i in range(400):
        tt = i * 0.25
        x = 0.1 * tt
        d = sup.update(x, 0.0, 0.0, tt)
        if d.trigger:
            fired_at = (tt, d)
            break
    assert fired_at is not None
    tt, d = fired_at
    assert d.reason == "lag"
    assert d.lag_m > cfg.reach_radius
    # At 0.2 m/s plan speed the distance floor needs > 1.75 s of lag.
    assert d.lag_s > cfg.reach_radius / 0.2 - 1e-9


def test_trigger_is_latched_once_per_segment():
    sup = SupervisorCore()
    poses, t = straight()
    sup.set_plan(poses, t)
    n = 0
    for i in range(30):  # robot parked at the start: lag keeps growing
        n += sup.update(0.0, 0.0, 0.0, i * 1.0).trigger
    assert n == 1


def test_ratio_condition_blocks_when_previous_segment_lagged_as_much():
    cfg = SupervisorConfig(reach_radius=0.35)
    sup = SupervisorCore(cfg)
    poses, t = straight()
    sup.set_plan(poses, t)
    sup.prev_lag = 5.0           # previous segment closed 5 s late
    d = sup.update(0.0, 0.0, 0.0, 4.0)   # 4 s lag: > distance floor, < 2x previous
    assert d.lag_m > cfg.reach_radius and not d.trigger
    d = sup.update(0.0, 0.0, 0.0, 10.5)  # 10.5 s >= 2 * 5 s
    assert d.trigger and d.reason == "lag"


def test_wall_scaling_and_stop():
    sup = SupervisorCore()
    poses, t = straight()
    sup.set_plan(poses, t)
    assert sup.update(0, 0, 0, 0.0, clearance=1.0).speed_scale == 1.0
    d = sup.update(0, 0, 0, 0.1, clearance=0.325)
    assert d.speed_scale == pytest.approx(0.5) and not d.stop
    assert d.trigger and d.reason == "near_wall"
    d = sup.update(0, 0, 0, 0.2, clearance=0.15)
    assert d.stop and d.speed_scale == 0.0 and d.trigger and d.reason == "wall"
    d = sup.update(0, 0, 0, 0.3, clearance=0.10)
    assert d.stop and d.trigger
    assert sup.stops == 1  # one stop episode


def test_plan_without_times_uses_cruise_speed():
    sup = SupervisorCore(SupervisorConfig(cruise_speed=0.5))
    poses, _ = straight()
    sup.set_plan(poses)
    assert sup.t_plan[-1] == pytest.approx(20.0)


def test_body_clearance_in_corridor():
    # 1 m wide corridor along x, walls at y=-0.5 and y=+0.5 (one cell each), 5 cm cells.
    res = 0.05
    wall = np.zeros((40, 200), bool)
    wall[10, :] = True   # y = -0.5 with origin_y = -1.0
    wall[30, :] = True   # y = +0.5
    swd = SignedWallDistance.from_wall_mask(wall, res, -5.0, -1.0)
    centred = swd.body_clearance(0.0, 0.0, 0.0)
    # half-width 0.2925 m: the side is ~0.2 m from either wall.
    assert 0.15 <= centred <= 0.25
    shifted = swd.body_clearance(0.0, 0.15, 0.0)
    assert shifted < centred
    across = swd.body_clearance(0.0, 0.0, math.pi / 2)  # half-length 0.315 m across
    assert across < centred


# ---- S1 wiring pieces: rebase, re-join target, wall gate, buffer splice ----

from types import SimpleNamespace

from agx_planning.runtime_corrector.trajectory_buffer import PlaybackSample, TrajectoryBuffer
from agx_planning.supervisor.walls import rejoin_clear


def test_rejoin_target_is_next_waypoint_at_least_t_min_ahead():
    sup = SupervisorCore()
    poses, t = straight()          # waypoints every 10 samples, dt 0.25 s
    sup.set_plan(poses, t)
    assert sup.rejoin_target(0, 0.25, 0.0) == 10
    assert sup.rejoin_target(0, 0.25, 3.0) == 20     # 12 samples needed -> next wp
    assert sup.rejoin_target(195, 0.25, 3.0) == 200  # past the last -> final sample


def test_rebase_clears_lag_and_latch_but_keeps_prev_lag():
    sup = SupervisorCore()
    poses, t = straight()
    sup.set_plan(poses, t)
    sup.prev_lag = 1.0
    assert sup.update(0.0, 0.0, 0.0, 10.0).trigger  # parked 10 s: lag
    sup.rebase(10.0)
    d = sup.update(0.0, 0.0, 0.0, 10.25)
    assert d.lag_s == pytest.approx(0.25) and not d.trigger and d.prev_lag_s == 1.0


def _corridor():
    wall = np.zeros((40, 200), bool)
    wall[10, :] = True
    wall[30, :] = True
    return SignedWallDistance.from_wall_mask(wall, 0.05, -5.0, -1.0)


def _samples(ys, x0=0.0):
    return [PlaybackSample(0.0, 0.0, (x0 + 0.1 * i, y, 0.0)) for i, y in enumerate(ys)]


def test_rejoin_gate_blocks_paths_into_walls_but_lets_a_stopped_robot_back_off():
    swd = _corridor()
    stop = 0.15
    # Body clearance at y = 0, 0.1, 0.15 is 0.20, 0.15, 0.10 m (5 cm grid).
    assert rejoin_clear(_samples([0.0] * 5), swd, 0.2, stop)
    assert not rejoin_clear(_samples([0.0, 0.1, 0.15, 0.0]), swd, 0.2, stop)
    # Started at 0.10 m (a stopped robot), moving back to the centre: allowed.
    assert rejoin_clear(_samples([0.15, 0.1, 0.0]), swd, 0.10, stop)
    # ... but not if it ends inside the stop distance.
    assert not rejoin_clear(_samples([0.15, 0.15]), swd, 0.10, stop)


def _buffer(n=50):
    buf = TrajectoryBuffer(0.1)
    buf.reset(1, 0.1)
    chunk = SimpleNamespace(wheel_left=[1.0] * n, wheel_right=[1.0] * n,
                            pose_x=[0.1 * i for i in range(n)], pose_y=[0.0] * n,
                            pose_theta=[0.0] * n)
    buf.add_chunk(0, chunk)
    buf.mark_result(True)
    return buf


def test_buffer_retime_moves_time_cursor():
    buf = _buffer()
    for _ in range(10):
        buf.advance()
    buf.retime(4)
    assert buf.advance().pose[0] == pytest.approx(0.4)


def test_buffer_splice_serves_overlay_then_resumes_and_blocks_retime():
    buf = _buffer()
    buf.advance()
    rejoin = [PlaybackSample(2.0, 2.0, (9.0 + i, 1.0, 0.0)) for i in range(3)]
    buf.splice(rejoin, 30)
    assert buf.overlay_active
    buf.retime(0)                 # ignored while the overlay plays
    got = [buf.advance().pose[0] for _ in range(4)]
    assert got == [9.0, 10.0, 11.0, pytest.approx(3.0)]
    assert not buf.overlay_active and not buf.is_done()
    while buf.advance() is not None:
        pass
    assert buf.is_done()


def test_relative_thresholds_follow_the_plan_clearance():
    sup = SupervisorCore()
    poses, t = straight()
    c_plan = np.full(len(poses), 2.0)
    c_plan[:20] = 0.2                       # a corridor at the start
    sup.set_plan(poses, t, clearance=c_plan)
    assert sup.wall_thresholds(0) == pytest.approx((0.15, 0.05))
    assert sup.wall_thresholds(100) == pytest.approx((0.5, 0.15))   # open space: agreed values
    d = sup.update(0.0, 0.0, 0.0, 0.0, clearance=0.2)   # on plan in the corridor
    assert d.speed_scale == 1.0 and not d.trigger and not d.stop
    d = sup.update(0.0, 0.0, 0.0, 0.1, clearance=0.04)
    assert d.stop and d.reason == "wall"
    # A plan that itself overlaps a wall still stops at contact_dist.
    sup.set_plan(poses, t, clearance=np.full(len(poses), -0.1))
    assert sup.wall_thresholds(0)[1] == pytest.approx(0.03)


def test_buffer_rate_dilates_the_time_cursor():
    buf = _buffer()
    got = [buf.advance(rate=0.5).pose[0] for _ in range(4)]
    assert got == pytest.approx([0.0, 0.0, 0.1, 0.1])
    assert buf.advance(rate=0.0).pose[0] == pytest.approx(0.2)
    assert buf.advance(rate=0.0).pose[0] == pytest.approx(0.2)


def test_buffer_cancel_overlay_returns_to_the_plan_cursor():
    buf = _buffer()
    buf.advance()
    buf.advance()
    buf.splice([PlaybackSample(0, 0, (9.0, 1.0, 0.0))] * 5, 30)
    buf.advance()
    buf.cancel_overlay()
    assert not buf.overlay_active
    assert buf.advance().pose[0] == pytest.approx(0.2)


def test_offplan_trigger_fires_on_schedule_once_per_segment():
    # On schedule along x but 0.6 m to the side: lag never grows, the
    # cross-track error does (S1 re-smoke, 00219).
    sup = SupervisorCore()
    poses, t = straight()
    sup.set_plan(poses, t)
    reasons = [sup.update(poses[k, 0], 0.6, 0.0, t[k]).reason for k in range(5)]
    assert reasons[0] == "offplan" and reasons[1:] == [""] * 4
    # 0.4 m off does not fire; offplan_dist=0 disables it.
    for y, cfg in ((0.4, SupervisorConfig()), (0.6, SupervisorConfig(offplan_dist=0.0))):
        s = SupervisorCore(cfg)
        s.set_plan(poses, t)
        assert not any(s.update(poses[k, 0], y, 0.0, t[k]).trigger for k in range(5))
