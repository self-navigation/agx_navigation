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


def test_trigger_switches_drop_lag_and_near_wall_but_keep_slowing_and_stopping():
    sup = SupervisorCore(SupervisorConfig(lag_trigger=False, near_wall_trigger=False))
    poses, t = straight()
    sup.set_plan(poses, t)
    for i in range(30):  # parked: lag grows past every threshold
        assert not sup.update(0.0, 0.0, 0.0, i * 1.0).trigger
    d = sup.update(0, 0, 0, 30.1, clearance=0.325)
    assert d.speed_scale == pytest.approx(0.5) and not d.trigger
    d = sup.update(0, 0, 0, 30.2, clearance=0.15)
    assert d.stop and d.trigger and d.reason == "wall"
    # off-plan still fires
    sup2 = SupervisorCore(SupervisorConfig(lag_trigger=False, near_wall_trigger=False))
    sup2.set_plan(poses, t)
    d = sup2.update(0.0, 0.8, 0.0, 0.0)
    assert d.trigger and d.reason == "offplan"


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


def test_offplan_bound_per_sample_replaces_offplan_dist():
    # Covariance bound (scenario b): 0.3 m on the first half, 0.8 m after.
    poses, t = straight()
    bound = np.where(np.arange(len(poses)) < 100, 0.3, 0.8)
    sup = SupervisorCore()
    sup.set_plan(poses, t, offplan_bound=bound)
    d = sup.update(poses[10, 0], 0.4, 0.0, t[10])
    assert d.trigger and d.reason == "offplan" and d.offplan_at == pytest.approx(0.3)
    s2 = SupervisorCore()
    s2.set_plan(poses, t, offplan_bound=bound)
    # 0.6 m off where the bound is 0.8: no trigger, though offplan_dist is 0.5.
    assert not any(s2.update(poses[k, 0], 0.6, 0.0, t[k]).trigger for k in range(150, 155))
    with pytest.raises(ValueError):
        SupervisorCore().set_plan(poses, t, offplan_bound=bound[:-1])


def test_wall_layer_off_neither_slows_nor_stops_but_keeps_gate_thresholds():
    poses, t = straight()
    sup = SupervisorCore(SupervisorConfig(wall_layer=False))
    sup.set_plan(poses, t)
    d = sup.update(poses[5, 0], 0.0, 0.0, t[5], clearance=0.0)
    assert not d.stop and d.speed_scale == 1.0 and d.reason != "wall"
    assert sup.wall_thresholds(5)[1] == pytest.approx(SupervisorConfig().stop_dist)


# ---- Wall-crossing re-join gate (2026-10-10, job 244) ----

def _thin_wall():
    # One-cell wall across a 4 m x 4 m map, vertical at x in [0.5, 0.55), 5 cm cells.
    wall = np.zeros((80, 80), bool)
    wall[:, 50] = True             # origin_x = -2.0 -> x = 0.5
    return SignedWallDistance.from_wall_mask(wall, 0.05, -2.0, -2.0)


def _xs(xs, y=0.0, yaw=0.0):
    return [PlaybackSample(0.0, 0.0, (x, y, yaw)) for x in xs]


def test_hull_penetration_grows_through_a_thin_wall():
    swd = _thin_wall()
    assert swd.hull_penetration(-0.5, 0.0, 0.0) == 0.0
    edge = swd.hull_penetration(0.2, 0.0, 0.0)      # front edge 0.515: on the wall
    deep = swd.hull_penetration(0.5, 0.0, 0.0)      # wall under the body's middle
    assert 0.0 < edge < 0.06 and deep > 0.25


def test_rejoin_gate_rejects_a_path_through_a_wall_from_a_touching_start():
    swd = _thin_wall()
    c0 = swd.body_clearance(0.2, 0.0, 0.0)
    assert c0 < 0.0                                 # touching: the old gate was vacuous
    path = _xs(np.arange(0.2, 1.31, 0.1))
    assert not rejoin_clear(path, swd, c0, 0.03)


def test_rejoin_gate_lets_a_touching_robot_back_away():
    swd = _thin_wall()
    c0 = swd.body_clearance(0.2, 0.0, 0.0)
    path = _xs(np.arange(0.2, -0.61, -0.1))
    assert rejoin_clear(path, swd, c0, 0.03)


def test_rejoin_gate_rejects_touching_again_after_clearing():
    swd = _thin_wall()
    c0 = swd.body_clearance(0.2, 0.0, 0.0)
    # Back off, then drive back onto the wall, then away again to a clear end.
    xs = list(np.arange(0.2, -0.31, -0.1)) + [0.2, 0.0, -0.3, -0.6]
    assert not rejoin_clear(_xs(xs), swd, c0, 0.03)


def test_rejoin_gate_rejects_stepping_over_a_wall_between_samples():
    swd = _thin_wall()
    # Clear start well left of the wall, then one sample 1.3 m on: neither
    # sample overlaps, but the body passes over the wall in between.
    start, end = -0.5, 1.6
    assert swd.hull_penetration(start, 0.0, 0.0) == 0.0
    assert swd.hull_penetration(end, 0.0, 0.0) == 0.0
    c0 = swd.body_clearance(start, 0.0, 0.0)
    assert not rejoin_clear(_xs([start, end]), swd, c0, 0.03)
    # The same two samples without the wall pass.
    empty = SignedWallDistance.from_wall_mask(np.zeros((80, 80), bool) | _corner(), 0.05,
                                              -2.0, -2.0)
    assert rejoin_clear(_xs([start, end]), empty, empty.body_clearance(start, 0, 0), 0.03)


def _corner():
    m = np.zeros((80, 80), bool)
    m[0, 0] = True                 # one far-away wall cell so the EDT is finite
    return m


def test_interpolation_bounds_hull_motion_to_half_a_cell():
    swd = _thin_wall()
    fine = swd.interpolate([(0.0, 0.0, 0.0), (1.0, 0.0, math.pi / 2)])
    r = float(np.hypot(swd.body[:, 0], swd.body[:, 1]).max())
    d = np.hypot(np.diff(fine[:, 0]), np.diff(fine[:, 1])) + np.abs(np.diff(fine[:, 2])) * r
    assert d.max() <= 0.025 + 1e-9
    assert fine[-1] == pytest.approx([1.0, 0.0, math.pi / 2])


# ---- Off-plan trigger hold (2026-10-10) ----

def _offplan_sup(**kw):
    sup = SupervisorCore(SupervisorConfig(lag_trigger=False, near_wall_trigger=False, **kw))
    poses, t = straight()
    sup.set_plan(poses, t)
    return sup


def test_offplan_hold_defaults_off():
    sup = _offplan_sup()
    d = sup.update(0.0, 0.8, 2.0, 0.0)           # 2 rad off the plan's heading
    assert d.trigger and d.reason == "offplan" and d.held == ""
    assert sup.offplan_held == 0


def test_offplan_hold_on_heading_error_then_fires_when_aligned():
    sup = _offplan_sup(offplan_max_heading=0.5)
    for i in range(3):
        d = sup.update(0.0, 0.8, 1.0, 0.1 * i)
        assert not d.trigger and d.held == "heading"
        assert d.heading_err == pytest.approx(1.0)
    assert sup.offplan_held == 1                 # one episode, not one per tick
    d = sup.update(0.0, 0.8, 0.2, 0.3)           # turned back towards the plan
    assert d.trigger and d.reason == "offplan" and d.held == ""
    # Heading wraps: yaw 2pi - 0.1 is 0.1 rad off, not held.
    sup2 = _offplan_sup(offplan_max_heading=0.5)
    assert sup2.update(0.0, 0.8, 2 * math.pi - 0.1, 0.0).trigger


def test_offplan_hold_counts_separate_episodes():
    sup = _offplan_sup(offplan_max_heading=0.5)
    sup.update(0.0, 0.8, 1.0, 0.0)
    sup.update(0.0, 0.0, 0.0, 0.1)               # back on the plan: not off-plan, not held
    sup.update(0.0, 0.8, -1.0, 0.2)
    assert sup.offplan_held == 2


def test_offplan_hold_in_the_plan_tail():
    sup = _offplan_sup(offplan_hold_tail_frac=0.1)
    # Plan is 201 samples along x in [0, 10]: the last 10% starts at k >= 180.9.
    sup.k_star = 185
    d = sup.update(9.25, 0.8, 0.0, 46.0)
    assert not d.trigger and d.held == "tail" and sup.offplan_held == 1
    sup2 = _offplan_sup(offplan_hold_tail_frac=0.1)
    sup2.k_star = 100
    d = sup2.update(5.0, 0.8, 0.0, 25.0)
    assert d.trigger and d.reason == "offplan"


def test_offplan_hold_leaves_wall_stops_alone():
    sup = SupervisorCore(SupervisorConfig(offplan_max_heading=0.1, offplan_hold_tail_frac=1.0))
    poses, t = straight()
    sup.set_plan(poses, t)
    d = sup.update(0.0, 0.8, 1.0, 0.0, clearance=0.0)
    assert d.stop and d.trigger and d.reason == "wall"


def test_buffer_last_index_tracks_plan_samples_and_marks_overlay():
    buf = _buffer()
    assert buf.last_index == -1
    buf.advance()
    buf.advance()
    assert buf.last_index == 1
    buf.splice([PlaybackSample(0.0, 0.0, (9.0, 1.0, 0.0))], 30)
    buf.advance()
    assert buf.last_index == -1
    buf.advance()
    assert buf.last_index == 30
