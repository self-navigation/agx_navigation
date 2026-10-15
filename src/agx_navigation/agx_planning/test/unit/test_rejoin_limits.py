"""Tests for re-join wheel-speed limits and landing windows.

shooting_solver._ode_effort carries the planner's soft speed barriers,
solve_rejoin rejects an over-limit solution as "infeasible" instead of letting
rejoin_service clip it, and RejoinSolver keeps the cheapest feasible landing
window that passes the caller's gate. Pure: no map, no field, no ROS.
"""

import sys

import numpy as np
import pytest

_rclpy_mock = sys.modules.pop("rclpy", None)
try:
    from agx_planning.pmp_planner import PlannerConfig, PMPShootingSolver
    from agx_planning.pmp_planner.rejoin_service import RejoinSolver
    from agx_planning.runtime_corrector.trajectory_buffer import PlaybackSample
finally:
    if _rclpy_mock is not None:
        sys.modules["rclpy"] = _rclpy_mock

GOAL = np.array([20.0, 0.0, 0.0])
INF = float("inf")


def _cfg(**kw):
    """A low wheel limit (8 rad/s = 0.64 m/s straight) so the cases stay
    small; the v/omega barriers off unless asked, to isolate the wheel one."""
    base = dict(mode="offline", w_wheel_max=8.0, rejoin_w_v_barrier=0.0,
                rejoin_w_omega_barrier=0.0)
    base.update(kw)
    return PlannerConfig(**base)


def _cruise(cfg, x, v=0.3):
    wl, wr = cfg.body_to_wheels(v, 0.0)
    return np.array([x, 0.0, 0.0, wl, wr])


def _solve(cfg, dist, T_w):
    s = PMPShootingSolver(cfg, None)
    return s.solve_rejoin(_cruise(cfg, 0.0), _cruise(cfg, dist), T_w, GOAL, cost="effort")


def test_wheel_barrier_keeps_speeds_inside_the_limit():
    """2.2 m in 4 s from and to 0.3 m/s: the barrier-free minimum-effort
    profile peaks at ~8.46 rad/s, over the 8 rad/s limit; a profile inside it
    exists (flatter, longer at speed), and the barrier finds it."""
    free = _solve(_cfg(rejoin_w_wheel_barrier=0.0, rejoin_wheel_tol=INF), 2.2, 4.0)
    assert free["success"] and free["max_wheel"] > 8.3
    r = _solve(_cfg(), 2.2, 4.0)
    assert r["success"], r["message"]
    assert r["max_wheel"] <= 8.0 + 0.01
    np.testing.assert_allclose(r["sol"](4.0)[0:5], _cruise(_cfg(), 2.2), atol=1e-3)
    # the barrier costs effort: the constrained optimum is dearer
    assert r["J"] >= free["J"]


def test_barrier_inactive_below_the_limit_changes_nothing():
    on = _solve(_cfg(), 2.0, 4.0)
    off = _solve(_cfg(rejoin_w_wheel_barrier=0.0), 2.0, 4.0)
    assert on["success"] and off["success"] and on["max_wheel"] < 8.0
    t = np.linspace(0.0, 4.0, 41)
    np.testing.assert_allclose(on["sol"](t)[0:5], off["sol"](t)[0:5], atol=1e-6)


@pytest.mark.parametrize("barrier", [0.0, 50.0])
def test_over_limit_solution_is_reported_infeasible(barrier):
    """3 m in 4 s needs ~10 rad/s at peak even with the barrier: converged,
    but not drivable, so it is a failure with a reason, not a clipped answer."""
    r = _solve(_cfg(rejoin_w_wheel_barrier=barrier), 3.0, 4.0)
    assert not r["success"]
    assert r["status"] == "infeasible"
    assert "exceeds w_wheel_max" in r["message"]
    assert r["max_wheel"] > 8.5
    assert "sol" not in r and "sol_last" in r


def test_feasibility_check_can_be_disabled():
    r = _solve(_cfg(rejoin_w_wheel_barrier=0.0, rejoin_wheel_tol=INF), 3.0, 4.0)
    assert r["success"] and r["max_wheel"] > 8.0


def test_v_barrier_has_the_planner_form():
    """With the v barrier on, a cruise that the pins allow below v_max is
    pulled under it: 2.2 m in 4 s at v_max = 0.6 m/s."""
    r = _solve(_cfg(rejoin_w_wheel_barrier=0.0, rejoin_w_v_barrier=50.0, v_max=0.6), 2.2, 4.0)
    free = _solve(_cfg(rejoin_w_wheel_barrier=0.0, v_max=0.6), 2.2, 4.0)
    y, yf = r["sol"](np.linspace(0, 4, 201)), free["sol"](np.linspace(0, 4, 201))
    v = 0.5 * 0.08 * (y[3] + y[4])
    vf = 0.5 * 0.08 * (yf[3] + yf[4])
    assert vf.max() > 0.66 and v.max() < vf.max() - 0.03


# ---- RejoinSolver: rejection and landing windows --------------------------

def _plan(cfg, n=150, v=0.3, dt=0.1):
    wl, wr = cfg.body_to_wheels(v, 0.0)
    return [PlaybackSample(left=wl, right=wr, pose=(v * dt * i, 0.0, 0.0)) for i in range(n)]


def test_service_returns_none_on_infeasible_instead_of_clipping():
    cfg = _cfg()
    rs = RejoinSolver(_plan(cfg), None, cfg, dt=0.1)
    x0 = rs.plan[10] - np.array([1.2, 0.0, 0.0, 0.0, 0.0])  # 1.2 m behind
    assert rs.solve(x0, 40, k_now=10) is None
    assert rs.last["status"] == "infeasible"
    assert rs.last["windows"][0]["status"] == "infeasible"


def test_landing_candidates():
    cfg = _cfg()
    rs = RejoinSolver(_plan(cfg, n=100), None, cfg, dt=0.1, windows=(3.0, 5.0, 8.0))
    # target at 4 s: the 3 s window collapses onto it; 8 s clips to the end
    assert rs.landing_candidates(50, 10) == [(50, 40), (60, 50), (90, 80)]
    assert rs.landing_candidates(30, 25) == [(30, 30), (55, 30), (75, 50), (99, 74)]
    # near the end every window clips to the last index: one candidate,
    # stretched to T_w_min
    assert rs.landing_candidates(99, 95) == [(99, 30)]
    assert RejoinSolver(_plan(cfg), None, cfg, dt=0.1).landing_candidates(40, 10) == [(40, 30)]


def test_windows_pick_the_cheapest_feasible_one_and_report_the_landing():
    """1.2 m behind a 0.3 m/s plan: catching up in 3 s needs > 8 rad/s
    (infeasible), 5 s and 8 s do not. The answer lands where the cheapest
    feasible window does, and its last sample IS the plan there."""
    cfg = _cfg()
    rs = RejoinSolver(_plan(cfg), None, cfg, dt=0.1, windows=(3.0, 5.0, 8.0))
    x0 = rs.plan[10] - np.array([1.2, 0.0, 0.0, 0.0, 0.0])
    out = rs.solve(x0, 40, k_now=10)
    assert out is not None, rs.last
    w = {round(c["T_w"], 1): c for c in rs.last["windows"]}
    assert w[3.0]["status"] == "infeasible"
    ok = [c for c in rs.last["windows"] if c["passed"]]
    assert len(ok) >= 1
    best = min(ok, key=lambda c: c["cost"])
    assert rs.last["target_index"] == best["tgt"]
    land = rs.last["target_index"]
    assert len(out) == land - 10 + 1   # on schedule: resume at land + 1
    np.testing.assert_allclose([*out[-1].pose, out[-1].left, out[-1].right],
                               rs.plan[land], atol=1e-3)
    assert max(max(abs(s.left), abs(s.right)) for s in out) <= 8.0 + cfg.rejoin_wheel_tol
    # costs are on one horizon: each = own J + plan playback to the farthest
    far = max(c["tgt"] for c in rs.last["windows"])
    for c in ok:
        assert c["J_tail"] == pytest.approx(rs._plan_cum[far] - rs._plan_cum[c["tgt"]])


def test_gate_selects_among_windows():
    """A gate that rejects the cheapest window makes the next cheapest win;
    the gate is called with the candidate's landing index."""
    cfg = _cfg()
    rs = RejoinSolver(_plan(cfg), None, cfg, dt=0.1, windows=(3.0, 5.0, 8.0))
    x0 = rs.plan[10] + np.array([0.0, 0.3, 0.0, 0.0, 0.0])
    assert rs.solve(x0, 40, k_now=10) is not None
    ranked = sorted((c for c in rs.last["windows"] if c["passed"]), key=lambda c: c["cost"])
    assert len(ranked) == 3
    seen = []

    def gate(smp, t):
        seen.append((t, len(smp)))
        return t != ranked[0]["tgt"]

    out = rs.solve(x0, 40, k_now=10, accept=gate, wall_retry=False)
    assert out is not None and rs.last["target_index"] == ranked[1]["tgt"]
    assert sorted(seen) == sorted((c["tgt"], c["tgt"] - 10 + 1) for c in ranked)
    # every window rejected: the cheapest blocked answer comes back, for the
    # caller's own gate to count as blocked
    out = rs.solve(x0, 40, k_now=10, accept=lambda smp, t: False, wall_retry=False)
    assert out is not None and rs.last["status"] == "blocked"
    assert rs.last["target_index"] == ranked[0]["tgt"]


def test_no_windows_is_the_single_target_behaviour():
    cfg = _cfg()
    rs = RejoinSolver(_plan(cfg), None, cfg, dt=0.1)
    x0 = rs.plan[10] + np.array([0.0, 0.2, 0.0, 0.0, 0.0])
    out = rs.solve(x0, 40, k_now=10)
    assert out is not None and len(out) == 31 and rs.last["target_index"] == 40
    assert len(rs.last["windows"]) == 1


def test_parse_windows():
    from agx_planning.supervisor.core import SupervisorConfig, parse_windows
    assert parse_windows(SupervisorConfig().rejoin_windows) == (3.0, 5.0, 8.0)
    assert parse_windows("3") == (3.0,)
    assert parse_windows("3, 6") == (3.0, 6.0)


def test_a_pin_over_the_limit_is_not_blamed_on_the_solver():
    """x0's MEASURED wheels already at 9 rad/s (limit 8): a solution that only
    slows down from there is feasible; the bar is the pin's own speed."""
    cfg = _cfg()
    s = PMPShootingSolver(cfg, None)
    x0 = np.array([0.0, 0.0, 0.0, 9.0, 9.0])
    r = s.solve_rejoin(x0, _cruise(cfg, 1.5), 4.0, GOAL, cost="effort")
    assert r["success"], r["message"]
    assert 8.0 < r["max_wheel"] <= 9.0 + 1e-3
