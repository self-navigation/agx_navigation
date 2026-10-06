"""SupervisorCore (#41): waypoints, per-segment lag, trigger rule, wall scaling."""

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
